# Imported for their registration side effects; the out-of-tree
# RowParallelLinear in `linear` only affects layers built after the import, so
# this module must be imported before any model is constructed.
from vllm_torchtpu.layers.adapter import attention as attention
from vllm_torchtpu.layers.adapter import linear as linear
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.tokenizers import register_tokenizers


def register_layers() -> None:
    register_models()
    register_tokenizers()
    _maybe_install_frontend_tokcache()
    _maybe_install_frontend_fix()


def _maybe_install_frontend_tokcache() -> None:
    import os

    if os.environ.get("QX_TOKENIZE_PREFIX_CACHE", "1") != "1":
        return
    try:
        from vllm_torchtpu.frontend_tokcache import (
            maybe_install_tokenize_prefix_cache,
        )

        maybe_install_tokenize_prefix_cache()
    except Exception as e:  # never break startup
        import logging

        logging.getLogger(__name__).warning(
            "[QX_TOKCACHE] install failed: %r", e
        )


def _maybe_install_frontend_fix() -> None:
    import os

    if os.environ.get("QX_LAZY_REASONING_COUNT", "1") != "1":
        return
    try:
        from vllm_torchtpu.frontend_fix import maybe_install_lazy_reasoning_count

        maybe_install_lazy_reasoning_count()
    except Exception as e:  # never break startup
        import logging

        logging.getLogger(__name__).warning(
            "[QX_LAZY_REASONING_COUNT] install failed: %r", e
        )
