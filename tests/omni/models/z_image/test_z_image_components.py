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

    def test_z_image_splash_attention_compiles_without_graph_breaks(self):
        """Splash path traces into one graph under torch.compile(backend="tpu").

        The DiT blocks are compiled regionally; the splash custom op must be
        reachable by dynamo (op name as a trace-time constant resolved through
        ``torch.ops.pallas``) or every block would graph-break / fall back to
        eager. Uses the main-block geometry (S=4128 padded to 4224).
        """
        from vllm_torchtpu.omni.attention import can_use_splash_attention

        torch.manual_seed(0)
        num_heads, head_size, seq_len = 30, 128, 4128
        impl = TpuSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=1.0 / math.sqrt(head_size),
        )
        q, k, v = (
            torch.randn(1, seq_len, num_heads, head_size, dtype=torch.bfloat16).to(
                "tpu"
            )
            for _ in range(3)
        )
        assert can_use_splash_attention(q, k, v, None, False)

        eager_out = impl.forward(q, k, v)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(impl.forward)(q, k, v)
        assert explanation.graph_break_count == 0, explanation.break_reasons
        assert explanation.graph_count == 1

        torch._dynamo.reset()
        compiled = torch.compile(impl.forward, backend="tpu", dynamic=False)
        compiled_out = compiled(q, k, v)
        assert compiled_out.shape == eager_out.shape
        torch.testing.assert_close(
            compiled_out.float().cpu(), eager_out.float().cpu(), atol=0.0, rtol=0.0
        )

    def test_z_image_unified_prepare_auto_pad_canary(self):
        """Canary: TPU patch enables ``auto_pad=True`` on ``unified_prepare``.

        ``patchify_and_embed`` already pads image and caption sequences to
        ``SEQ_MULTI_OF=32``, and ``_patch_z_image_transformer_block_flush``
        flushes each ``ZImageTransformerBlock`` so a single XLA module only
        contains that block's 4 ``all_to_all_single`` collectives instead of
        accumulating 120+ across all 30 blocks. Marking ``unified_prepare``
        with ``auto_pad=True`` lets ``UlyssesParallelAttention`` skip its 240
        per-layer ``_all_gather_int`` host syncs in ``advanced_uaa`` mode
        (``ulysses_degree=4`` and ``ulysses_degree=8``).
        """
        from vllm_omni.diffusion.models.z_image.z_image_transformer import (
            ZImageTransformer2DModel,
        )

        from vllm_torchtpu.omni.patches import _patch_z_image_transformer_block_flush

        _patch_z_image_transformer_block_flush()
        unified_prepare_plan = ZImageTransformer2DModel._sp_plan["unified_prepare"]
        assert isinstance(unified_prepare_plan, dict)
        assert unified_prepare_plan, "unified_prepare SP plan unexpectedly empty"
        for idx, sp_input in unified_prepare_plan.items():
            assert sp_input.auto_pad is True, (
                f"_sp_plan['unified_prepare'][{idx}].auto_pad was not promoted to True"
            )

    def test_z_image_encode_prompt_canary(self):
        """Canary: upstream ``_encode_prompt`` still matches the TPU mirror.

        ``_patch_z_image_prompt_encoder_tpu`` re-implements the tokenization of
        ``ZImagePipeline._encode_prompt`` and replaces its text-encoder call by a
        compiled program that yields ``hidden_states[-2]``. If upstream changes
        any of the mirrored details, this test fails so the patch is revisited.
        """
        import inspect

        from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline

        from vllm_torchtpu.omni.patches import _unwrap_tpu_patch

        src = inspect.getsource(_unwrap_tpu_patch(ZImagePipeline._encode_prompt))
        # Whitespace-insensitive view for the multi-line chat-message literal.
        src_flat = " ".join(src.split())
        assert '[{"role": "user", "content": prompt_item}]' in src_flat.replace(
            "[ {", "[{"
        ).replace("}, ]", "}]"), (
            "upstream ZImagePipeline._encode_prompt changed (chat message literal); "
            "update _patch_z_image_prompt_encoder_tpu"
        )
        for needle in (
            "add_generation_prompt=True",
            "enable_thinking=True",
            'padding="max_length"',
            "max_length=max_sequence_length",
            "truncation=True",
            "output_hidden_states=True",
            ".hidden_states[-2]",
            "prompt_embeds[i][prompt_masks[i]]",
        ):
            assert needle in src, (
                f"upstream ZImagePipeline._encode_prompt changed ({needle!r} missing); "
                "update _patch_z_image_prompt_encoder_tpu"
            )

    def test_compiled_prompt_encoder_matches_hf_hidden_states(self):
        """``_TpuCompiledPromptEncoder`` == HF ``hidden_states[-2]`` on TPU."""
        pytest.importorskip("transformers")
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from vllm_torchtpu.omni.patches import _TpuCompiledPromptEncoder

        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            attn_implementation="sdpa",
        )
        model = Qwen3ForCausalLM(config).to(torch.float32).to("tpu").eval()
        input_ids = torch.randint(0, 128, (2, 16)).to("tpu")
        attention_mask = torch.tensor([[1] * 10 + [0] * 6, [1] * 16], dtype=torch.long)

        with torch.no_grad():
            ref = model(
                input_ids=input_ids,
                attention_mask=attention_mask.to("tpu").bool(),
                output_hidden_states=True,
            ).hidden_states[-2]
        out = _TpuCompiledPromptEncoder(model)(input_ids, attention_mask)

        assert out.shape == ref.shape == (2, 16, 64)
        torch.testing.assert_close(
            out[0, :10].cpu(), ref[0, :10].cpu(), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(out[1].cpu(), ref[1].cpu(), atol=1e-4, rtol=1e-4)

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
