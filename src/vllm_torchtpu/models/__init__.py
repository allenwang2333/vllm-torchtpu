# SPDX-License-Identifier: Apache-2.0
"""TPU model definitions provided by vllm_torchtpu.

``register_models`` is the ``vllm.general_plugins`` entry point (see
pyproject.toml). vLLM loads general plugins from ``EngineArgs`` *before*
``create_model_config()``, which is the constraint that matters: the architecture
mapping has to exist before ``ModelConfig`` resolves the checkpoint's
architecture.

The platform plugin entry point runs earlier but cannot be used for this — at
that point ``vllm.utils.torch_utils`` is only partially initialized and importing
``ModelRegistry`` raises a circular import.
"""

_registered = False


def register_models() -> None:
    """vLLM general-plugin entry point: register TPU model definitions.

    Idempotent, and never fatal — one model failing to register must not stop
    the plugin from serving models that do not need it.
    """
    global _registered
    if _registered:
        return
    _registered = True

    from vllm_torchtpu.logger import init_logger
    logger = init_logger(__name__)

    try:
        from vllm_torchtpu.models import kimi_k3
        kimi_k3.register()
    except Exception as error:
        logger.warning("Could not register TPU Kimi model: %s",
                       error,
                       exc_info=True)


__all__ = ["register_models"]
