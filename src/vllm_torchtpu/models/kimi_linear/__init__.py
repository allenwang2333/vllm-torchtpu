# SPDX-License-Identifier: Apache-2.0
"""TPU support for Kimi-Linear.

``register()`` installs the TPU KDA attention layer as an out-of-tree
``PluggableLayer``, replacing vLLM's ``KimiGatedDeltaNetAttention`` wherever the
in-tree ``KimiLinearForCausalLM`` constructs it. No model architecture is
registered -- see ``model.py`` for why.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

_registered = False


def register() -> None:
    """Idempotently install the TPU KDA layer for Kimi-Linear."""
    global _registered
    if _registered:
        return
    _registered = True

    from vllm_torchtpu.models.kimi_linear.kda import _register

    cls = _register()
    logger.info(
        "Registered TPU Kimi-Linear KDA layer: %s replaces in-tree "
        "KimiGatedDeltaNetAttention (chunked Pallas prefill / recurrent decode)",
        cls.__name__)


__all__ = ["register"]
