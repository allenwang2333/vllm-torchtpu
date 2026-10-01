# SPDX-License-Identifier: Apache-2.0
"""Model-specific patches for Gemma4 multimodal models on TPU."""

from typing import TYPE_CHECKING

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_GEMMA4_MM_ARCHITECTURES = {
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
}

_TEXT_ONLY_ARCHITECTURES = {
    "Gemma4ForCausalLM",
}


def patch_gemma4_mm_model_classes() -> None:
    """Set supports_encoder_tp_data = True on Gemma4 multimodal model classes."""
    try:
        from vllm.model_executor.models.gemma4_mm import (
            Gemma4ForConditionalGeneration,
        )

        if not getattr(
            Gemma4ForConditionalGeneration, "supports_encoder_tp_data", False
        ):
            Gemma4ForConditionalGeneration.supports_encoder_tp_data = True
            logger.info(
                "Applied TPU patch: enabled supports_encoder_tp_data on "
                "Gemma4ForConditionalGeneration."
            )
    except (ImportError, AttributeError):
        pass

    try:
        from vllm.model_executor.models.gemma4_unified import (
            Gemma4UnifiedForConditionalGeneration,
        )

        if not getattr(
            Gemma4UnifiedForConditionalGeneration, "supports_encoder_tp_data", False
        ):
            Gemma4UnifiedForConditionalGeneration.supports_encoder_tp_data = True
            logger.info(
                "Applied TPU patch: enabled supports_encoder_tp_data on "
                "Gemma4UnifiedForConditionalGeneration."
            )
    except (ImportError, AttributeError):
        pass


def _is_gemma4_mm_model(model_config: "ModelConfig | None") -> bool:
    """Check if the provided ModelConfig corresponds to a Gemma4 multimodal model."""
    if model_config is None:
        return False

    architectures = getattr(model_config, "architectures", []) or []
    if any(arch in _GEMMA4_MM_ARCHITECTURES for arch in architectures):
        return True
    if any(arch in _TEXT_ONLY_ARCHITECTURES for arch in architectures):
        return False

    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        hf_archs = getattr(hf_config, "architectures", []) or []
        if any(arch in _GEMMA4_MM_ARCHITECTURES for arch in hf_archs):
            return True
        if any(arch in _TEXT_ONLY_ARCHITECTURES for arch in hf_archs):
            return False
        model_type = getattr(hf_config, "model_type", "")
        if model_type == "gemma4":
            return hasattr(hf_config, "vision_config")

    model_name = str(getattr(model_config, "model", "")).lower()
    is_gemma4 = "gemma-4" in model_name or "gemma4" in model_name
    has_mm = getattr(model_config, "multimodal_config", None) is not None
    return is_gemma4 and has_mm


def maybe_patch_gemma4_mm(model_config: "ModelConfig | None" = None) -> bool | None:
    """Apply TPU patches for Gemma4 multimodal models.

    Enables data-parallel multimodal vision encoding when --mm-encoder-tp-mode
    data is configured by setting supports_encoder_tp_data = True on the model
    classes and updating model_config._model_info.
    """
    if not _is_gemma4_mm_model(model_config):
        return False

    patch_gemma4_mm_model_classes()

    if model_config is not None:
        model_info = getattr(model_config, "_model_info", None)
        if model_info is not None:
            object.__setattr__(
                model_info,
                "supports_multimodal_encoder_tp_data",
                True,
            )

    return None
