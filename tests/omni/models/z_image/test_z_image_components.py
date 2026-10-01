# SPDX-License-Identifier: Apache-2.0
"""Component-level numerical correctness tests for Z-Image on TPU hardware.

Tested Z-Image Components:
==========================
1. RopeEmbedder (vllm_omni) + RotaryEmbedding / apply_rope_to_qk: 3D RoPE
   (theta=256, axes_dims=[32, 48, 48]) and Q/K rotation parity on TPU vs CPU.
2. TpuSDPAImpl (vllm_torchtpu): diffusion attention parity at Z-Image head
   geometry (30 heads x 128 head_dim) against float64 CPU ground truth.
"""

import math

import pytest
import torch

pytest.importorskip("vllm_omni", reason="vllm_omni is required for Z-Image tests")

from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk
from vllm_omni.diffusion.models.z_image.z_image_transformer import RopeEmbedder

from vllm_torchtpu.omni.attention import TpuSDPAImpl

pytestmark = [pytest.mark.omni]


def _assert_numerical_parity(
    actual: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    min_cos_sim: float = 0.999,
) -> None:
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


class TestZImageComponentsOnTpu:
    """Component-level numerical parity tests for Z-Image on TPU (dim=3840)."""

    def test_z_image_rope_embedder_and_rotation_parity(self):
        """Test Z-Image 3D RopeEmbedder (theta=256, axes_dims=[32, 48, 48]) and
        apply_rope_to_qk on TPU vs CPU ground truth."""
        torch.manual_seed(0)
        batch_size = 1
        seq_len = 128
        num_heads = 30
        head_size = 128

        embedder_cpu = RopeEmbedder(
            theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1024, 512, 512]
        )
        embedder_tpu = RopeEmbedder(
            theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1024, 512, 512]
        )

        ids_cpu = torch.stack(
            [
                torch.arange(seq_len, dtype=torch.int32),
                torch.arange(seq_len, dtype=torch.int32) % 64,
                torch.arange(seq_len, dtype=torch.int32) // 64,
            ],
            dim=-1,
        )
        cos_cpu, sin_cpu = embedder_cpu(ids_cpu)
        cos_tpu, sin_tpu = embedder_tpu(ids_cpu.to("tpu"))

        assert cos_tpu.device.type == "tpu" and sin_tpu.device.type == "tpu"
        assert cos_tpu.shape == sin_tpu.shape == (seq_len, head_size // 2)
        torch.testing.assert_close(cos_tpu.cpu(), cos_cpu, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sin_tpu.cpu(), sin_cpu, atol=1e-5, rtol=1e-5)

        rope = RotaryEmbedding(is_neox_style=False)
        q_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.float32
        )
        k_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.float32
        )

        ref_q, ref_k = apply_rope_to_qk(
            rope, q_cpu, k_cpu, (cos_cpu.unsqueeze(0), sin_cpu.unsqueeze(0))
        )
        tpu_q, tpu_k = apply_rope_to_qk(
            rope,
            q_cpu.to("tpu"),
            k_cpu.to("tpu"),
            (cos_tpu.unsqueeze(0), sin_tpu.unsqueeze(0)),
        )

        assert tpu_q.device.type == "tpu" and tpu_k.device.type == "tpu"
        _assert_numerical_parity(tpu_q.cpu(), ref_q, atol=1e-4, rtol=1e-4)
        _assert_numerical_parity(tpu_k.cpu(), ref_k, atol=1e-4, rtol=1e-4)

    def test_z_image_sdpa_shape_parity_on_tpu(self):
        """Test TpuSDPAImpl at Z-Image head geometry (30 heads x 128 head_dim)."""
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 256
        num_heads = 30
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = TpuSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
        )
        q_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )
        k_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )
        v_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )

        probs = (
            torch.einsum("bqhd,bkhd->bhqk", q_cpu.double(), k_cpu.double()) * scale
        ).softmax(dim=-1)
        ref_out = torch.einsum("bhqk,bkhd->bqhd", probs, v_cpu.double())

        tpu_out = impl.forward(q_cpu.to("tpu"), k_cpu.to("tpu"), v_cpu.to("tpu"))
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, seq_len, num_heads, head_size)
        _assert_numerical_parity(tpu_out.cpu(), ref_out, atol=1e-2, rtol=1e-2)

    def test_z_image_unified_prepare_auto_pad_canary(self):
        """Canary: upstream must keep ``auto_pad=False`` on ``unified_prepare``.

        ``patchify_and_embed`` already pads image and caption sequences to
        ``SEQ_MULTI_OF=32``, so ``auto_pad`` is a no-op numerically. It is not a
        no-op for scheduling: ``auto_pad=True`` marks local sequence lengths as
        equal, ``UlyssesParallelAttention`` then skips its per-layer sync, and
        the resulting run of async ``all_to_all_single`` collectives in one lazy
        PJRT graph deadlocks at ``ulysses_degree=8`` on TPU. If upstream flips
        this flag, the multi-chip path needs re-validation before bumping the pin.
        """
        from vllm_omni.diffusion.models.z_image.z_image_transformer import (
            ZImageTransformer2DModel,
        )

        unified_prepare_plan = ZImageTransformer2DModel._sp_plan["unified_prepare"]
        assert isinstance(unified_prepare_plan, dict)
        assert unified_prepare_plan, "unified_prepare SP plan unexpectedly empty"
        for idx, sp_input in unified_prepare_plan.items():
            assert sp_input.auto_pad is False, (
                f"_sp_plan['unified_prepare'][{idx}].auto_pad changed upstream; "
                "re-validate Z-Image ulysses_degree=8 on TPU before accepting."
            )

    @pytest.mark.parametrize(
        ("ulysses_degree", "expected_mode"),
        [(8, "advanced_uaa"), (4, "advanced_uaa"), (2, "strict"), (1, "strict")],
    )
    def test_ulysses_mode_promotion_on_real_parallel_config(
        self, ulysses_degree: int, expected_mode: str
    ):
        """Run the platform hook against vLLM-Omni's real ``DiffusionParallelConfig``.

        Guards the attribute names the hook relies on (``ulysses_degree``,
        ``ulysses_mode``) against upstream renames: with the real dataclass a
        rename would surface here instead of silently disabling the promotion.
        """
        from types import SimpleNamespace

        from vllm_omni.diffusion.data import DiffusionParallelConfig

        from vllm_torchtpu.omni.platform import OmniTpuPlatform

        parallel_config = DiffusionParallelConfig(ulysses_degree=ulysses_degree)
        assert parallel_config.ulysses_mode == "strict", "upstream default changed"
        od_config = SimpleNamespace(
            model_class_name="ZImagePipeline", parallel_config=parallel_config
        )
        OmniTpuPlatform.init_diffusion_model_runner_runtime(
            vllm_config=None, od_config=od_config, device=torch.device("cpu")
        )
        assert parallel_config.ulysses_mode == expected_mode
