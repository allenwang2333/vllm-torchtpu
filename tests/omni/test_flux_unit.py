# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness and component-wise unit tests for FLUX models on TPU hardware.

Tested FLUX Components:
=======================
1. TpuSDPAImpl (vllm_torchtpu): TPU SDPA diffusion attention backend parity for standard,
   asymmetric joint text/image, and GQA head expansion against CPU ground truth.
2. FluxPosEmbed / Flux2PosEmbed (vllm_omni): Multi-coordinate RoPE frequency calculation,
   float32 precision, and cos^2 + sin^2 == 1.0 trigonometric invariant on TPU.
3. RotaryEmbedding / apply_rope_to_qk (vllm_omni): RoPE attention rotation kernel application
   on TPU vs CPU ground truth with bitwise replay determinism.
"""

import math

import pytest
import torch
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk
from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2PosEmbed
from vllm_omni.diffusion.models.flux.flux_transformer import FluxPosEmbed

from vllm_torchtpu.omni.attention import TpuSDPABackend, TpuSDPAImpl
from vllm_torchtpu.omni.patches import (apply_omni_model_specific_patches,
                                        apply_omni_tpu_patches)

pytestmark = [pytest.mark.omni]


@pytest.fixture(scope="module", autouse=True)
def setup_omni_patches():
    apply_omni_tpu_patches()
    apply_omni_model_specific_patches()


def _assert_numerical_parity(
    actual: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    min_cos_sim: float = 0.999,
):
    """Assert cosine similarity and floating-point tolerance bounds."""
    a_flat = actual.flatten().float()
    r_flat = reference.flatten().float()

    norm_a = torch.linalg.norm(a_flat)
    norm_r = torch.linalg.norm(r_flat)
    if norm_a > 0 and norm_r > 0:
        cos_sim = float(torch.dot(a_flat, r_flat) / (norm_a * norm_r))
        assert cos_sim >= min_cos_sim, (
            f"Cosine similarity {cos_sim:.5f} fell below minimum required {min_cos_sim}"
        )

    torch.testing.assert_close(
        actual.float(),
        reference.float(),
        atol=atol,
        rtol=rtol,
        msg="TPU kernel output diverged from CPU baseline",
    )


class TestFluxTpuSDPAAttention:
    """Numerical correctness and parity tests for TPU SDPA attention backend.

    vllm_omni delegates all FLUX diffusion attention to this backend on TPU.
    We test standard, asymmetric cross-attention, and GQA configurations directly
    against CPU execution.

    Architectural Note on Why We Do Not Call vllm_omni Attention() directly:
    -----------------------------------------------------------------------
    `vllm_omni.diffusion.attention.layer.Attention` queries the active platform via
    `current_omni_platform.get_diffusion_attn_backend_cls()`. On CPU environments,
    current_omni_platform is `UnspecifiedOmniPlatform`, which has no attention backend
    registered and raises:
        `NotImplementedError: No forward implementation for platform: <UnspecifiedOmniPlatform>`
    `TpuSDPAImpl` is the hardware implementation that vllm-torchtpu registers into vllm_omni.
    Because TpuSDPAImpl wraps PyTorch's native SDPA, we test its forward pass directly
    on CPU as the exact mathematical baseline against TPU execution.
    """

    def test_backend_registration_and_properties(self):
        assert TpuSDPABackend.get_name() == "TPU_SDPA"
        assert TpuSDPABackend.get_impl_cls() is TpuSDPAImpl
        assert TpuSDPABackend.supports_attention_mask() is True

    def test_flux_standard_attention_numeric_parity_on_tpu(self):
        """Test FLUX.1 standard attention numeric parity against CPU baseline."""
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 64
        num_heads = 24
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = TpuSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
        )

        query_cpu = torch.randn(batch_size,
                                seq_len,
                                num_heads,
                                head_size,
                                dtype=torch.float32)
        key_cpu = torch.randn(batch_size,
                              seq_len,
                              num_heads,
                              head_size,
                              dtype=torch.float32)
        value_cpu = torch.randn(batch_size,
                                seq_len,
                                num_heads,
                                head_size,
                                dtype=torch.float32)

        # Baseline execution on CPU using the exact same component
        ref_out = impl.forward(query_cpu, key_cpu, value_cpu)

        # TPU execution in bfloat16
        q_tpu = query_cpu.to(device="tpu", dtype=torch.bfloat16)
        k_tpu = key_cpu.to(device="tpu", dtype=torch.bfloat16)
        v_tpu = value_cpu.to(device="tpu", dtype=torch.bfloat16)

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, seq_len, num_heads, head_size)
        assert tpu_out.dtype == torch.bfloat16
        assert torch.all(torch.isfinite(tpu_out.cpu()))

        # Bitwise determinism on TPU (exact replay parity)
        tpu_out_replay = impl.forward(q_tpu, k_tpu, v_tpu)
        torch.testing.assert_close(
            tpu_out.cpu(),
            tpu_out_replay.cpu(),
            atol=0.0,
            rtol=0.0,
            msg="TPU execution must be bitwise identical for identical inputs",
        )

        # Numerical parity against CPU baseline (CosSim >= 0.999, bfloat16 tolerance)
        _assert_numerical_parity(tpu_out.cpu(), ref_out, atol=1e-2, rtol=1e-2)

    def test_flux_joint_attention_asymmetry_numeric_parity_on_tpu(self):
        """Test asymmetric joint text/image cross-attention numeric parity on TPU."""
        torch.manual_seed(123)
        batch_size = 2
        q_len = 128
        kv_len = 64
        num_heads = 24
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = TpuSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
        )

        query_cpu = torch.randn(batch_size,
                                q_len,
                                num_heads,
                                head_size,
                                dtype=torch.float32)
        key_cpu = torch.randn(batch_size,
                              kv_len,
                              num_heads,
                              head_size,
                              dtype=torch.float32)
        value_cpu = torch.randn(batch_size,
                                kv_len,
                                num_heads,
                                head_size,
                                dtype=torch.float32)

        ref_out = impl.forward(query_cpu, key_cpu, value_cpu)

        q_tpu = query_cpu.to(device="tpu", dtype=torch.bfloat16)
        k_tpu = key_cpu.to(device="tpu", dtype=torch.bfloat16)
        v_tpu = value_cpu.to(device="tpu", dtype=torch.bfloat16)

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, q_len, num_heads, head_size)
        assert torch.all(torch.isfinite(tpu_out.cpu()))

        _assert_numerical_parity(tpu_out.cpu(), ref_out, atol=1e-2, rtol=1e-2)

    def test_flux_gqa_repeat_interleave_numeric_parity_on_tpu(self):
        """Test grouped-query attention repeat-interleave numeric parity on TPU."""
        torch.manual_seed(456)
        batch_size = 1
        seq_len = 32
        num_heads = 24
        num_kv_heads = 8
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = TpuSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
            num_kv_heads=num_kv_heads,
        )

        query_cpu = torch.randn(batch_size,
                                seq_len,
                                num_heads,
                                head_size,
                                dtype=torch.float32)
        key_cpu = torch.randn(batch_size,
                              seq_len,
                              num_kv_heads,
                              head_size,
                              dtype=torch.float32)
        value_cpu = torch.randn(batch_size,
                                seq_len,
                                num_kv_heads,
                                head_size,
                                dtype=torch.float32)

        ref_out = impl.forward(query_cpu, key_cpu, value_cpu)

        q_tpu = query_cpu.to(device="tpu", dtype=torch.bfloat16)
        k_tpu = key_cpu.to(device="tpu", dtype=torch.bfloat16)
        v_tpu = value_cpu.to(device="tpu", dtype=torch.bfloat16)

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, seq_len, num_heads, head_size)
        assert torch.all(torch.isfinite(tpu_out.cpu()))

        _assert_numerical_parity(tpu_out.cpu(), ref_out, atol=1e-2, rtol=1e-2)


class TestFluxRoPEPositionEmbeddingPrecision:
    """Verify vllm_omni RoPE coordinate embeddings maintain numerical parity and float32 precision on TPU."""

    def test_flux_pos_embed_numeric_parity_and_invariants_on_tpu(self):
        """Test 3D RoPE coordinates parity against CPU ground truth and trigonometric invariants."""
        axes_dim = [16, 56, 56]
        theta = 10000
        seq_len = 32

        # Directly imported from vllm_omni
        embedder_cpu = FluxPosEmbed(theta=theta, axes_dim=axes_dim)
        ids_cpu = torch.stack(
            [
                torch.zeros(seq_len),
                torch.arange(seq_len),
                torch.arange(seq_len),
            ],
            dim=-1,
        )

        ref_cos, ref_sin = embedder_cpu(ids_cpu)

        embedder_tpu = FluxPosEmbed(theta=theta, axes_dim=axes_dim)
        ids_tpu = ids_cpu.to(device="tpu")
        cos_tpu, sin_tpu = embedder_tpu(ids_tpu)

        expected_dim = sum(axes_dim) // 2
        assert cos_tpu.device.type == "tpu"
        assert sin_tpu.device.type == "tpu"
        assert cos_tpu.shape == (seq_len, expected_dim)
        assert sin_tpu.shape == (seq_len, expected_dim)

        # Trigonometric invariant: cos^2 + sin^2 must equal 1.0 on TPU within float32 tolerance
        unity = cos_tpu.float()**2 + sin_tpu.float()**2
        torch.testing.assert_close(
            unity.cpu(),
            torch.ones_like(unity.cpu()),
            atol=1e-4,
            rtol=1e-4,
            msg="RoPE cos^2 + sin^2 must equal 1.0 on TPU",
        )

        # Numerical parity against CPU ground truth (float32 tolerance)
        torch.testing.assert_close(
            cos_tpu.cpu().float(),
            ref_cos.float(),
            atol=1e-4,
            rtol=1e-4,
            msg="TPU cos embeddings diverged from CPU reference",
        )
        torch.testing.assert_close(
            sin_tpu.cpu().float(),
            ref_sin.float(),
            atol=1e-4,
            rtol=1e-4,
            msg="TPU sin embeddings diverged from CPU reference",
        )

        # Bounded in [-1, 1]
        assert (cos_tpu.cpu() >= -1.0 - 1e-4).all() and (cos_tpu.cpu()
                                                         <= 1.0 + 1e-4).all()
        assert (sin_tpu.cpu() >= -1.0 - 1e-4).all() and (sin_tpu.cpu()
                                                         <= 1.0 + 1e-4).all()

    def test_flux_rope_float32_numeric_parity_on_tpu(self):
        """Ensure FLUX.1 RoPE position embeddings produce tensors with CPU parity."""
        torch.manual_seed(789)
        embedder_cpu = FluxPosEmbed(theta=10000, axes_dim=[16, 32, 32])
        ids = torch.randint(0, 50, (10, 3))
        ref_cos, ref_sin = embedder_cpu(ids)

        embedder_tpu = FluxPosEmbed(theta=10000, axes_dim=[16, 32, 32])
        cos_tpu, sin_tpu = embedder_tpu(ids.to("tpu"))

        assert cos_tpu.device.type == "tpu"
        assert sin_tpu.device.type == "tpu"
        assert cos_tpu.shape == (10, sum([16, 32, 32]) // 2)
        assert cos_tpu.dtype == ref_cos.dtype
        assert sin_tpu.dtype == ref_sin.dtype

        torch.testing.assert_close(cos_tpu.cpu(),
                                   ref_cos,
                                   atol=1e-4,
                                   rtol=1e-4)
        torch.testing.assert_close(sin_tpu.cpu(),
                                   ref_sin,
                                   atol=1e-4,
                                   rtol=1e-4)

    def test_flux2_rope_float32_numeric_parity_on_tpu(self):
        """Ensure FLUX.2 RoPE position embeddings produce tensors with CPU parity."""
        torch.manual_seed(101112)
        embedder_cpu = Flux2PosEmbed(theta=2000, axes_dim=[32, 32, 32, 32])
        ids = torch.randint(0, 50, (8, 4))
        ref_cos, ref_sin = embedder_cpu(ids)

        embedder_tpu = Flux2PosEmbed(theta=2000, axes_dim=[32, 32, 32, 32])
        cos_tpu, sin_tpu = embedder_tpu(ids.to("tpu"))

        assert cos_tpu.device.type == "tpu"
        assert sin_tpu.device.type == "tpu"
        assert cos_tpu.shape == (8, sum([32, 32, 32, 32]) // 2)
        assert cos_tpu.dtype == ref_cos.dtype
        assert sin_tpu.dtype == ref_sin.dtype

        torch.testing.assert_close(cos_tpu.cpu(),
                                   ref_cos,
                                   atol=1e-4,
                                   rtol=1e-4)
        torch.testing.assert_close(sin_tpu.cpu(),
                                   ref_sin,
                                   atol=1e-4,
                                   rtol=1e-4)


class TestFluxRoPEKernelApplication:
    """Numerical correctness and parity tests for RoPE kernel application.

    Directly imports RotaryEmbedding and apply_rope_to_qk from
    vllm_omni.diffusion.layers.rope, which is the exact module and function
    invoked inside FluxAttention.forward() to rotate Q and K tensors before attention.
    """

    def test_apply_rope_to_qk_numeric_parity_on_tpu(self):
        """Test apply_rope_to_qk parity between CPU baseline and TPU hardware."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 32
        num_heads = 24
        head_size = 128
        axes_dim = [16, 56, 56]

        rope = RotaryEmbedding(is_neox_style=False)
        pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dim)

        ids_cpu = torch.stack(
            [
                torch.zeros(seq_len),
                torch.arange(seq_len),
                torch.arange(seq_len),
            ],
            dim=-1,
        )
        cos_cpu, sin_cpu = pos_embed(ids_cpu)

        q_cpu = torch.randn(batch_size,
                            seq_len,
                            num_heads,
                            head_size,
                            dtype=torch.float32)
        k_cpu = torch.randn(batch_size,
                            seq_len,
                            num_heads,
                            head_size,
                            dtype=torch.float32)

        ref_q, ref_k = apply_rope_to_qk(rope, q_cpu, k_cpu, (cos_cpu, sin_cpu))

        q_tpu = q_cpu.to(device="tpu")
        k_tpu = k_cpu.to(device="tpu")
        cos_tpu = cos_cpu.to(device="tpu")
        sin_tpu = sin_cpu.to(device="tpu")

        tpu_q, tpu_k = apply_rope_to_qk(rope, q_tpu, k_tpu, (cos_tpu, sin_tpu))

        assert tpu_q.device.type == "tpu"
        assert tpu_k.device.type == "tpu"
        assert tpu_q.shape == q_cpu.shape
        assert tpu_k.shape == k_cpu.shape
        assert torch.all(torch.isfinite(tpu_q.cpu()))
        assert torch.all(torch.isfinite(tpu_k.cpu()))

        # Bitwise replay determinism
        tpu_q_replay, tpu_k_replay = apply_rope_to_qk(rope, q_tpu, k_tpu,
                                                      (cos_tpu, sin_tpu))
        torch.testing.assert_close(tpu_q.cpu(),
                                   tpu_q_replay.cpu(),
                                   atol=0.0,
                                   rtol=0.0)
        torch.testing.assert_close(tpu_k.cpu(),
                                   tpu_k_replay.cpu(),
                                   atol=0.0,
                                   rtol=0.0)

        # Numerical parity against CPU ground truth
        _assert_numerical_parity(tpu_q.cpu(), ref_q, atol=1e-4, rtol=1e-4)
        _assert_numerical_parity(tpu_k.cpu(), ref_k, atol=1e-4, rtol=1e-4)
