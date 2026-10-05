# Imported for their registration side effects; the out-of-tree
# RowParallelLinear in `linear` only affects layers built after the import, so
# this module must be imported before any model is constructed.
from vllm_torchtpu.layers.adapter import attention as attention
from vllm_torchtpu.layers.adapter import linear as linear
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.tokenizers import register_tokenizers

logger = init_logger(__name__)


def register_layers() -> None:
    register_models()
    register_tokenizers()
    _install_frontend_patches()


def _install_frontend_patches() -> None:
    # API-server patches. vLLM loads general plugins when it parses the CLI
    # and builds engine arguments, before the renderer and the OpenAI
    # serving objects exist. A failure here leaves stock vLLM behaviour.
    try:
        from vllm_torchtpu.frontend.lazy_reasoning_count import (
            maybe_install_lazy_reasoning_count,
        )

        maybe_install_lazy_reasoning_count()
    except Exception:
        logger.warning(
            "Lazy reasoning count install failed; chat streaming keeps vLLM's "
            "per-delta reasoning-token recount.",
            exc_info=True,
        )
    try:
        from vllm_torchtpu.frontend.tokenize_prefix_cache import (
            maybe_install_tokenize_prefix_cache,
        )

        maybe_install_tokenize_prefix_cache()
    except Exception:
        logger.warning(
            "Tokenize prefix cache install failed; prompts are tokenized in full.",
            exc_info=True,
        )
