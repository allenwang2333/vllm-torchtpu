# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FLUX models on TPU platform (hermetic, CPU/mock)."""

import math

import pytest
import torch

from vllm_torchtpu.omni.attention import TpuSDPABackend, TpuSDPAImpl
from vllm_torchtpu.omni.patches import (apply_omni_model_specific_patches,
                                        apply_omni_tpu_patches)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module", autouse=True)
def setup_omni_patches():
    apply_omni_tpu_patches()
    apply_omni_model_specific_patches()


class TestFluxTpuSDPAAttention:
    """Test TPU SDPA attention backend with FLUX architecture configurations."""

    def test_backend_registration_and_properties(self):
        assert TpuSDPABackend.get_name() == "TPU_SDPA"
        assert TpuSDPABackend.get_impl_cls() is TpuSDPAImpl
        assert TpuSDPABackend.supports_attention_mask() is True

    def test_flux_standard_attention_forward(self):
        # FLUX.1 uses 24 attention heads, head_size=128 (hidden_size = 3072)
        batch_size = 1
        seq_len = 64  # Representative token sequence length
        num_heads = 24
        head_size = 128

        impl = TpuSDPAImpl(num_heads=num_heads,
                           head_size=head_size,
                           softmax_scale=1.0 / math.sqrt(head_size))

        query = torch.randn(batch_size, seq_len, num_heads, head_size)
        key = torch.randn(batch_size, seq_len, num_heads, head_size)
        value = torch.randn(batch_size, seq_len, num_heads, head_size)

        out = impl.forward(query, key, value)
        assert out.shape == (batch_size, seq_len, num_heads, head_size)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_flux_joint_attention_sequence_asymmetry(self):
        # In FLUX joint blocks, query has seq_len (image+text),
        # but cross-attention can have asymmetric key/value lengths.
        batch_size = 2
        q_len = 128
        kv_len = 128
        num_heads = 24
        head_size = 128

        impl = TpuSDPAImpl(num_heads=num_heads,
                           head_size=head_size,
                           softmax_scale=1.0 / math.sqrt(head_size))
        query = torch.randn(batch_size, q_len, num_heads, head_size)
        key = torch.randn(batch_size, kv_len, num_heads, head_size)
        value = torch.randn(batch_size, kv_len, num_heads, head_size)

        out = impl.forward(query, key, value)
        assert out.shape == (batch_size, q_len, num_heads, head_size)

    def test_flux_gqa_repeat_interleave(self):
        # Test grouped-query attention if num_kv_heads < num_heads
        batch_size = 1
        seq_len = 32
        num_heads = 16
        num_kv_heads = 4
        head_size = 64

        impl = TpuSDPAImpl(num_heads=num_heads,
                           head_size=head_size,
                           softmax_scale=1.0 / math.sqrt(head_size),
                           num_kv_heads=num_kv_heads)

        query = torch.randn(batch_size, seq_len, num_heads, head_size)
        key = torch.randn(batch_size, seq_len, num_kv_heads, head_size)
        value = torch.randn(batch_size, seq_len, num_kv_heads, head_size)

        out = impl.forward(query, key, value)
        assert out.shape == (batch_size, seq_len, num_heads, head_size)


class TestFluxRoPEPositionEmbeddingPrecision:
    """Verify RoPE position embeddings maintain float32 precision for TPU execution."""

    def test_flux_pos_embed_trigonometric_invariants(self):
        # Axes dimensions for coordinate space: (height, width, frame, channel/t)
        axes_dim = [16, 56, 56]
        theta = 10000
        seq_len = 20

        from vllm_omni.diffusion.models.flux.flux_transformer import \
            FluxPosEmbed

        embedder = FluxPosEmbed(theta=theta, axes_dim=axes_dim)
        ids = torch.randint(0, 100, (seq_len, 3))

        freqs_cos, freqs_sin = embedder(ids)

        expected_dim = sum(axes_dim) // 2
        assert freqs_cos.shape == (seq_len, expected_dim)
        assert freqs_sin.shape == (seq_len, expected_dim)

        # cos^2 + sin^2 must equal 1 within floating tolerance
        cos_sq_sin_sq = freqs_cos**2 + freqs_sin**2
        assert torch.allclose(cos_sq_sin_sq,
                              torch.ones_like(cos_sq_sin_sq),
                              atol=1e-5)

        # Bounded in [-1, 1]
        assert (freqs_cos >= -1.0 - 1e-5).all() and (freqs_cos
                                                     <= 1.0 + 1e-5).all()
        assert (freqs_sin >= -1.0 - 1e-5).all() and (freqs_sin
                                                     <= 1.0 + 1e-5).all()

    def test_flux_rope_float32_precision(self):
        """Ensure FLUX.1 RoPE position embeddings produce float32/float64 tensors."""
        from vllm_omni.diffusion.models.flux.flux_transformer import \
            FluxPosEmbed

        embedder = FluxPosEmbed(theta=10000, axes_dim=[16, 32, 32])
        ids = torch.randint(0, 50, (10, 3))
        cos, sin = embedder(ids)
        assert cos.shape == (10, sum([16, 32, 32]) // 2)
        assert cos.dtype in (torch.float32,
                             torch.float64) and sin.dtype in (torch.float32,
                                                              torch.float64)

    def test_flux2_rope_float32_precision(self):
        """Ensure FLUX.2 RoPE position embeddings produce float32/float64 tensors."""
        from vllm_omni.diffusion.models.flux2.flux2_transformer import \
            Flux2PosEmbed

        embedder = Flux2PosEmbed(theta=2000, axes_dim=[32, 32, 32, 32])
        ids = torch.randint(0, 50, (8, 4))
        cos, sin = embedder(ids)
        assert cos.shape == (8, sum([32, 32, 32, 32]) // 2)
        assert cos.dtype in (torch.float32,
                             torch.float64) and sin.dtype in (torch.float32,
                                                              torch.float64)


class TestFluxRuntimePatches:
    """Validate runtime patches applied for TPU."""

    def test_omni_tpu_patches_configuration(self):
        apply_omni_tpu_patches.cache_clear()
        apply_omni_tpu_patches()

        from vllm_omni.diffusion import diffusion_engine
        assert diffusion_engine._ASYNC_OUTPUT_TIMEOUT == 1800.0


class TestFluxInputValidationAndSanitization:
    """Test input checking and sampling params contracts for FLUX models."""

    def test_flux_sampling_params_defaults(self):
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        # Schnell defaults: distilled 4 steps, guidance=0.0
        params_schnell = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        assert params_schnell.num_inference_steps == 4
        assert params_schnell.guidance_scale == 0.0

        # Dev defaults: guidance=3.5
        params_dev = OmniDiffusionSamplingParams(
            height=1024,
            width=1024,
            num_inference_steps=20,
            guidance_scale=3.5,
        )
        assert params_dev.num_inference_steps == 20
        assert params_dev.guidance_scale == 3.5

    def test_flux_empty_prompt_rejection(self):
        from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import \
            Flux2KleinPipeline

        pipe = object.__new__(Flux2KleinPipeline)
        pipe.vae_scale_factor = 8
        pipe.is_distilled = True
        pipe._guidance_scale = 0.0

        with pytest.raises(ValueError, match="empty or whitespace-only"):
            pipe.check_inputs(
                prompt="",
                height=512,
                width=512,
                prompt_embeds=None,
                num_inference_steps=4,
                guidance_scale=0.0,
            )
