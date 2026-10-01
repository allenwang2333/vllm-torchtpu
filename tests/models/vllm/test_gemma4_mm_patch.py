# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma4 multimodal patch and mm-encoder-tp-mode support on TPU."""

from dataclasses import dataclass
from types import SimpleNamespace

try:
    import pytest

    pytestmark = pytest.mark.cpu_test
except ImportError:
    pytest = None

from vllm_torchtpu.models.vllm.gemma4_mm_patch import (
    _is_gemma4_mm_model,
    maybe_patch_gemma4_mm,
    patch_gemma4_mm_model_classes,
)
from vllm_torchtpu.patch_registry import PATCHES


@dataclass(frozen=True)
class DummyModelInfo:
    supports_multimodal: bool = True
    supports_multimodal_encoder_tp_data: bool = False


def test_is_gemma4_mm_model_detection():
    # None config
    assert not _is_gemma4_mm_model(None)

    # Non-Gemma model
    qwen_config = SimpleNamespace(
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        hf_config=SimpleNamespace(model_type="qwen2_5_vl"),
        model="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    assert not _is_gemma4_mm_model(qwen_config)

    # Text-only Gemma4 model
    text_gemma = SimpleNamespace(
        architectures=["Gemma4ForCausalLM"],
        hf_config=SimpleNamespace(model_type="gemma4"),
        model="google/gemma-4-26b-it",
    )
    assert not _is_gemma4_mm_model(text_gemma)

    # Gemma4 multimodal model architectures
    mm_gemma = SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        hf_config=SimpleNamespace(
            model_type="gemma4",
            architectures=["Gemma4ForConditionalGeneration"],
            vision_config=SimpleNamespace(hidden_size=1152),
        ),
        model="google/gemma-4-26b-it",
    )
    assert _is_gemma4_mm_model(mm_gemma)

    unified_gemma = SimpleNamespace(
        architectures=["Gemma4UnifiedForConditionalGeneration"],
        hf_config=SimpleNamespace(
            model_type="gemma4",
            architectures=["Gemma4UnifiedForConditionalGeneration"],
        ),
        model="google/gemma-4-unified",
    )
    assert _is_gemma4_mm_model(unified_gemma)

    # Config with only hf_config.vision_config
    vision_only_gemma = SimpleNamespace(
        architectures=[],
        hf_config=SimpleNamespace(
            model_type="gemma4",
            vision_config=SimpleNamespace(hidden_size=1152),
        ),
        model="custom-gemma4",
    )
    assert _is_gemma4_mm_model(vision_only_gemma)


def test_patch_gemma4_mm_model_classes():
    # Verify idempotent class patching
    patch_gemma4_mm_model_classes()
    patch_gemma4_mm_model_classes()

    try:
        from vllm.model_executor.models.gemma4_mm import (
            Gemma4ForConditionalGeneration,
        )

        assert getattr(
            Gemma4ForConditionalGeneration, "supports_encoder_tp_data", False
        )
    except ImportError:
        pass


def test_maybe_patch_gemma4_mm_returns_false_for_non_matching_model():
    non_gemma_config = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        hf_config=SimpleNamespace(model_type="llama"),
        model="meta-llama/Llama-3-8B",
    )
    assert maybe_patch_gemma4_mm(non_gemma_config) is False
    assert maybe_patch_gemma4_mm(None) is False


def test_maybe_patch_gemma4_mm_applies_successfully():
    model_info = DummyModelInfo(supports_multimodal_encoder_tp_data=False)
    multimodal_config = SimpleNamespace(mm_encoder_tp_mode="weights")
    model_config = SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        hf_config=SimpleNamespace(model_type="gemma4"),
        model="google/gemma-4-26b-it",
        _model_info=model_info,
        multimodal_config=multimodal_config,
    )

    result = maybe_patch_gemma4_mm(model_config)
    assert result is None

    # Model info should be updated even if frozen
    assert model_config._model_info.supports_multimodal_encoder_tp_data is True


def test_patch_registry_contains_gemma4_mm_patch():
    targets = [p.target for p in PATCHES]
    assert "vllm_torchtpu.models.vllm.gemma4_mm_patch:maybe_patch_gemma4_mm" in targets

    patch = next(
        p
        for p in PATCHES
        if p.target == "vllm_torchtpu.models.vllm.gemma4_mm_patch:maybe_patch_gemma4_mm"
    )
    assert patch.stages == ("platform_activation", "model_load")
    assert patch.model_config is True
