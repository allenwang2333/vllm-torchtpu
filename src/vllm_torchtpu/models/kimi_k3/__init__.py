# SPDX-License-Identifier: Apache-2.0
"""Registration for the TPU Kimi model definition.

Step 1 registers a single architecture pointing at the dummy-logits skeleton in
``model.py``. A lazy ``"<module>:<class>"`` string is used so importing the
plugin does not import torch-heavy model code.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

ARCH_TO_CLASS = {
    "KimiLinearForCausalLM":
    "vllm_torchtpu.models.kimi_k3.model:KimiLinearForCausalLM",
}

_registered = False


def register() -> None:
    """Idempotently register the TPU Kimi architectures with vLLM."""
    global _registered
    if _registered:
        return
    _registered = True

    from vllm.model_executor.models import ModelRegistry

    existing = set(ModelRegistry.get_supported_archs())
    for arch, target in ARCH_TO_CLASS.items():
        if arch in existing:
            # vLLM ships an in-tree KimiLinearForCausalLM. Overriding it is
            # intentional here -- it is how the skeleton gets driven, including
            # as the inner language model of KimiK3ForConditionalGeneration,
            # which requests this architecture by name. But it is global: any
            # Kimi-Linear checkpoint served from this environment now gets the
            # skeleton. Warn loudly rather than shadow silently.
            logger.warning(
                "Kimi TPU: OVERRIDING in-tree vLLM architecture %s with the TPU "
                "skeleton %s. All %s checkpoints in this environment will "
                "produce dummy output.", arch, target, arch)
        ModelRegistry.register_model(arch, target)
        logger.info("Registered TPU model: %s -> %s", arch, target)


__all__ = ["ARCH_TO_CLASS", "register"]
